
import asyncio
import json
import os
import time
from typing import Optional
from datetime import datetime, timezone
from uuid import uuid4

from wire_check.models import IntakeRequest, WorkflowResponse
from wire_check.agents.entity import resolve_entity
from wire_check.agents.sanctions import screen_entity
from wire_check.agents.fraud import run_fraud_checks
from wire_check.agents.decision import make_decision
from wire_check.config import AGENT_SEED
from wire_check.utils.logger import get_logger

logger = get_logger("orchestrator_agent")

try:
    from uagents import Agent, Context, Model, Protocol
    from uagents_core.contrib.protocols.chat import (
        ChatMessage,
        ChatAcknowledgement,
        TextContent,
        EndSessionContent,
        chat_protocol_spec,
    )
    HAS_UAGENTS = True
except ImportError:
    HAS_UAGENTS = False


def run_wire_check_workflow(intake: IntakeRequest) -> WorkflowResponse:
    """Run the local entity, sanctions, fraud and decision workflow."""
    start_time = time.time()

    logger.info(
        f"Starting workflow for {intake.company_name}, "
        f"amount={intake.amount}"
    )

    entity_res = resolve_entity(intake)
    sanctions_res = screen_entity(entity_res)
    fraud_res = run_fraud_checks(intake, entity_res)
    decision_res = make_decision(
        intake, entity_res, sanctions_res, fraud_res
    )

    logger.info(
        f"Workflow completed in "
        f"{(time.time() - start_time) * 1000:.1f}ms"
    )

    return WorkflowResponse(
        intake=intake,
        entity=entity_res,
        sanctions=sanctions_res,
        fraud=fraud_res,
        decision=decision_res,
    )


if HAS_UAGENTS:

    # Address of the remote Wire-Check-jev ACP agent.
    JEV_ADDRESS = os.getenv("WIRE_CHECK_JEV_ADDRESS", "").strip()
    JEV_TIMEOUT_SECONDS = 35

    class WireCheckQuery(Model):
        company_name: str
        country: str
        registration_number: Optional[str] = None
        iban: str
        bank_country: str
        invoice_email_domain: str
        amount: float
        recent_bank_change: bool = False

        # Additional fields needed by the verification API.
        website: Optional[str] = None
        beneficiary_name: Optional[str] = None
        payment_currency: str = "USD"
        invoice_text: Optional[str] = None

    class WireCheckResponse(Model):
        status: str
        decision: str
        evidence: list[str]
        callback_script: Optional[str] = None
        jev_result: Optional[str] = None

    wire_check_agent = Agent(
        name="wire_check_orchestrator",
        seed=AGENT_SEED,
        mailbox=True,
        port=8001,
    )

    # ACP protocol used to communicate with Wire-Check-jev.
    jev_protocol = Protocol(spec=chat_protocol_spec)

    # This prototype allows one in-flight ACP verification at a time.
    jev_lock = asyncio.Lock()
    jev_pending_future = None

    @jev_protocol.on_message(ChatMessage)
    async def handle_jev_chat(
        ctx: Context, sender: str, msg: ChatMessage
    ):
        global jev_pending_future

        # Accept replies only from the configured remote agent.
        if not JEV_ADDRESS or sender != JEV_ADDRESS:
            ctx.logger.info(
                f"Ignoring ACP message from unconfigured sender: {sender}"
            )
            return

        reply_text = "\n".join(
            item.text
            for item in msg.content
            if isinstance(item, TextContent)
        ).strip()

        if jev_pending_future is not None and not jev_pending_future.done():
            jev_pending_future.set_result(reply_text)
        else:
            ctx.logger.info(f"Unmatched ACP reply: {reply_text}")

    @jev_protocol.on_message(ChatAcknowledgement)
    async def handle_jev_ack(
        ctx: Context, sender: str, msg: ChatAcknowledgement
    ):
        ctx.logger.info(f"ACP acknowledgement received from {sender}")

    wire_check_agent.include(jev_protocol, publish_manifest=True)

    async def call_jev(ctx: Context, msg: WireCheckQuery) -> str:
        """Send supplier details to Jev and wait for its ACP response."""
        global jev_pending_future

        if not JEV_ADDRESS:
            raise RuntimeError(
                "WIRE_CHECK_JEV_ADDRESS is not configured."
            )

        # Do not invent missing supplier or beneficiary information.
        payload = {
            "supplierName": msg.company_name,
            "country": msg.country,
            "website": msg.website or "",
            "registrationNumber": msg.registration_number or "",
            "beneficiaryName": msg.beneficiary_name or "",
            "isBankDetailsNewOrChanged": msg.recent_bank_change,
            "paymentAmount": msg.amount,
            "paymentCurrency": msg.payment_currency,
            "invoiceText": msg.invoice_text or (
                f"IBAN: {msg.iban}\n"
                f"Bank country: {msg.bank_country}\n"
                f"Invoice email domain: {msg.invoice_email_domain}"
            ),
        }

        async with jev_lock:
            loop = asyncio.get_running_loop()
            jev_pending_future = loop.create_future()

            outgoing = ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(payload),
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            try:
                ctx.logger.info(
                    f"Sending ACP verification request to {JEV_ADDRESS}"
                )
                await ctx.send(JEV_ADDRESS, outgoing)

                return await asyncio.wait_for(
                    jev_pending_future,
                    timeout=JEV_TIMEOUT_SECONDS,
                )
            finally:
                jev_pending_future = None

    # Keep the custom query protocol for existing callers.
    wire_check_proto = Protocol(
        name="WireCheckWorkflow",
        version="1.0",
    )

    @wire_check_proto.on_message(
        model=WireCheckQuery,
        replies={WireCheckResponse},
    )
    async def handle_wire_check_query(
        ctx: Context, sender: str, msg: WireCheckQuery
    ):
        ctx.logger.info(
            f"Received query from {sender}: {msg.company_name}"
        )

        intake = IntakeRequest(
            company_name=msg.company_name,
            country=msg.country,
            registration_number=msg.registration_number,
            iban=msg.iban,
            bank_country=msg.bank_country,
            invoice_email_domain=msg.invoice_email_domain,
            amount=msg.amount,
            recent_bank_change=msg.recent_bank_change,
        )

        # Run the existing local multi-agent workflow.
        try:
            response = await asyncio.to_thread(
                run_wire_check_workflow, intake
            )

            decision = response.decision.decision
            evidence = list(response.decision.evidence)
            callback_script = response.decision.callback_script

        except Exception as exc:
            ctx.logger.exception(f"Local workflow failed: {exc}")
            await ctx.send(
                sender,
                WireCheckResponse(
                    status="error",
                    decision="HOLD",
                    evidence=[
                        "Local workflow failed; manual review required."
                    ],
                ),
            )
            return

        # Ask the remote Jev agent for its independent verification.
        jev_result = None
        status = "success"

        try:
            jev_result = await call_jev(ctx, msg)
            evidence.append(
                "Wire-Check-jev ACP response: " + jev_result
            )
        except Exception as exc:
            status = "partial"
            ctx.logger.warning(
                f"Jev verification unavailable: {exc}"
            )
            evidence.append(
                "Remote Jev verification unavailable. "
                "The local workflow result is returned; "
                "remote verification was not completed."
            )

        await ctx.send(
            sender,
            WireCheckResponse(
                status=status,
                decision=decision,
                evidence=evidence,
                callback_script=callback_script,
                jev_result=jev_result,
            ),
        )

    wire_check_agent.include(wire_check_proto)

    @wire_check_agent.on_event("startup")
    async def startup(ctx: Context):
        ctx.logger.info(
            f"Orchestrator address: {wire_check_agent.address}"
        )
        ctx.logger.info(
            f"Jev address configured: {bool(JEV_ADDRESS)}"
        )


if __name__ == "__main__":
    if HAS_UAGENTS:
        print("Starting Wire Check Orchestrator...")
        wire_check_agent.run()
    else:
        print("Install uagents and uagents-core dependencies first.")
