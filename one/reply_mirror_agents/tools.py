from __future__ import annotations

from collections import Counter
from datetime import timedelta
import statistics
from typing import Any

from config import AppConfig
from data_loader import DatasetBundle
from linking import (
    describe_counterparty,
    determine_focal_user,
    resolve_transaction_parties,
    transaction_city,
    transaction_counterparty_key,
)
from models import EvidenceBundle, MessageRecord, TransactionRecord, UserProfile
from rules import score_rule_based_risk
from tracing import TraceManager
from utils import haversine_km, merchant_hint, normalize_text


class FraudTools:
    def __init__(
        self,
        dataset: DatasetBundle,
        config: AppConfig,
        tracer: TraceManager,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.tracer = tracer

    def get_transaction_context(
        self,
        transaction_id: str,
        dataset_path: str | None = None,
    ) -> dict[str, Any]:
        with self.tracer.span(
            "get_transaction_context",
            as_type="tool",
            input_payload={"transaction_id": transaction_id, "dataset_path": dataset_path},
        ):
            transaction = self.dataset.transactions_by_id[transaction_id]
            return {
                "transaction_id": transaction.transaction_id,
                "transaction_type": transaction.transaction_type,
                "amount": transaction.amount,
                "timestamp": transaction.timestamp.isoformat(),
                "location": transaction.location,
                "location_city": transaction_city(transaction),
                "payment_method": transaction.payment_method or None,
                "sender_id": transaction.sender_id,
                "recipient_id": transaction.recipient_id,
                "sender_iban": transaction.sender_iban or None,
                "recipient_iban": transaction.recipient_iban or None,
                "balance_after": transaction.balance_after,
                "description": transaction.description,
                "merchant_hint": merchant_hint(
                    transaction.location,
                    transaction.description,
                    transaction.transaction_type,
                ),
            }

    def resolve_transaction_parties(
        self,
        transaction: TransactionRecord,
        users_data: dict[str, UserProfile] | None = None,
    ) -> dict[str, Any]:
        with self.tracer.span(
            "resolve_transaction_parties",
            as_type="tool",
            input_payload={"transaction_id": transaction.transaction_id},
        ):
            resolved = resolve_transaction_parties(
                transaction,
                users_data or self.dataset.users_by_iban,
            )
            focal_user, focal_role, counterparty_user = determine_focal_user(
                transaction,
                self.dataset.users_by_iban,
            )
            return {
                "sender_user": resolved["sender_user"].full_name
                if resolved["sender_user"]
                else None,
                "recipient_user": resolved["recipient_user"].full_name
                if resolved["recipient_user"]
                else None,
                "focal_user": focal_user.full_name if focal_user else None,
                "focal_role": focal_role,
                "counterparty_user": counterparty_user.full_name
                if counterparty_user
                else None,
                "counterparty_description": describe_counterparty(
                    transaction,
                    focal_role,
                    counterparty_user,
                ),
            }

    def get_user_location_context(
        self,
        user: UserProfile | None,
        transaction_time,
        locations_data: dict[str, list] | None = None,
        window_hours: int | None = None,
        transaction: TransactionRecord | None = None,
    ) -> dict[str, Any]:
        with self.tracer.span(
            "get_user_location_context",
            as_type="tool",
            input_payload={
                "user": user.full_name if user else None,
                "transaction_id": transaction.transaction_id if transaction else None,
            },
        ):
            if user is None or user.biotag is None or transaction is None:
                return {
                    "relevance": "low",
                    "plausible": None,
                    "nearby_locations": [],
                    "fraud_signals": [],
                    "legitimacy_signals": [],
                    "uncertainties": ["Location context unavailable for this transaction"],
                }

            transaction_type = transaction.transaction_type
            location_city = transaction_city(transaction)
            if transaction_type not in {"in-person payment", "withdrawal"} or not location_city:
                return {
                    "relevance": "low",
                    "plausible": None,
                    "nearby_locations": [],
                    "fraud_signals": [],
                    "legitimacy_signals": ["Location plausibility is not central for this transaction type"],
                    "uncertainties": [],
                }

            hours = window_hours or self.config.location_window_hours
            observations = (locations_data or self.dataset.locations_by_biotag).get(user.biotag, [])
            window = timedelta(hours=hours)
            nearby = [
                observation
                for observation in observations
                if observation.timestamp and abs(observation.timestamp - transaction_time) <= window
            ]
            nearby_cities = sorted({item.city for item in nearby})
            legitimacy_signals: list[str] = []
            fraud_signals: list[str] = []
            uncertainties: list[str] = []
            plausible: bool | None = None

            if nearby:
                same_city = [item for item in nearby if normalize_text(item.city) == normalize_text(location_city)]
                if same_city:
                    plausible = True
                    legitimacy_signals.append(
                        f"Nearby GPS observations place the user in {location_city} around the transaction time"
                    )
                else:
                    plausible = False
                    nearest = min(
                        nearby,
                        key=lambda item: abs(item.timestamp - transaction_time),
                    )
                    hours_gap = abs((nearest.timestamp - transaction_time).total_seconds()) / 3600.0
                    fraud_signals.append(
                        f"Transaction took place in {location_city} while nearby GPS evidence points to {nearest.city} {hours_gap:.1f}h away"
                    )
                    distance = haversine_km(
                        user.residence_lat,
                        user.residence_lng,
                        nearest.lat,
                        nearest.lng,
                    )
                    if distance > 80:
                        fraud_signals.append(
                            f"Observed travel context is inconsistent with routine city presence at roughly {distance:.0f} km from home"
                        )
            else:
                if normalize_text(location_city) == normalize_text(user.residence_city):
                    plausible = True
                    legitimacy_signals.append(
                        "No nearby GPS sample exists, but the transaction city matches the user's residence city"
                    )
                else:
                    uncertainties.append(
                        "No nearby GPS observation exists to verify the transaction city directly"
                    )

            return {
                "relevance": "high",
                "plausible": plausible,
                "nearby_locations": [
                    {
                        "timestamp": item.timestamp.isoformat(),
                        "city": item.city,
                        "lat": item.lat,
                        "lng": item.lng,
                    }
                    for item in nearby[:6]
                ],
                "nearby_cities": nearby_cities,
                "fraud_signals": fraud_signals,
                "legitimacy_signals": legitimacy_signals,
                "uncertainties": uncertainties,
            }

    def _message_matches_transaction(
        self,
        transaction: TransactionRecord,
        message: MessageRecord,
    ) -> bool:
        tx_terms = {
            token
            for token in normalize_text(
                " ".join([transaction.location, transaction.description, transaction.transaction_type])
            ).split()
            if len(token) > 3 and token not in {"payment", "transfer", "order", "account"}
        }
        message_text = normalize_text(" ".join([message.subject, message.body, message.sender]))
        return any(term in message_text for term in tx_terms)

    def get_user_message_context(
        self,
        user: UserProfile | None,
        transaction_time,
        sms_data: list | None = None,
        mails_data: list | None = None,
        lookback_days: int | None = None,
        transaction: TransactionRecord | None = None,
    ) -> dict[str, Any]:
        with self.tracer.span(
            "get_user_message_context",
            as_type="tool",
            input_payload={
                "user": user.full_name if user else None,
                "transaction_id": transaction.transaction_id if transaction else None,
            },
        ):
            if user is None or transaction is None:
                return {
                    "messages": [],
                    "recent_suspicious_count": 0,
                    "recent_legitimate_count": 0,
                    "fraud_signals": [],
                    "legitimacy_signals": [],
                    "uncertainties": ["Message context unavailable for this transaction"],
                }

            lower_bound = transaction_time - timedelta(days=lookback_days or self.config.message_lookback_days)
            upper_bound = transaction_time + timedelta(days=2)
            candidates = [
                item
                for item in self.dataset.messages_by_user_iban.get(user.iban, [])
                if item.timestamp and lower_bound <= item.timestamp <= upper_bound
            ]
            candidates.sort(key=lambda item: abs(item.timestamp - transaction_time))

            recent_suspicious = []
            recent_legitimate = []
            fraud_signals: list[str] = []
            legitimacy_signals: list[str] = []
            uncertainties: list[str] = []
            sampled_messages = []

            for message in candidates[:8]:
                matches_transaction = self._message_matches_transaction(transaction, message)
                sampled_messages.append(
                    {
                        "channel": message.channel,
                        "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                        "sender": message.sender,
                        "subject": message.subject,
                        "classifications": message.classifications,
                        "snippet": message.body[:220],
                        "matches_transaction": matches_transaction,
                        "domains": message.domains,
                    }
                )
                if "suspicious/phishing" in message.classifications:
                    recent_suspicious.append(message)
                    fraud_signals.append(
                        f"Recent {message.channel} includes phishing-style language from {message.sender or 'unknown sender'}"
                    )
                    if message.domains:
                        fraud_signals.append(
                            f"Recent communication references suspicious domains: {', '.join(message.domains[:3])}"
                        )
                if "security alert" in message.classifications and "suspicious/phishing" in message.classifications:
                    recent_suspicious.append(message)
                if "legitimate order/bill" in message.classifications and matches_transaction:
                    recent_legitimate.append(message)
                    legitimacy_signals.append(
                        f"A recent {message.channel} appears to support legitimate activity related to this merchant or transaction"
                    )

            if not candidates:
                uncertainties.append("No linked SMS or mail was found near the transaction date")

            return {
                "messages": sampled_messages,
                "recent_suspicious_count": len(recent_suspicious),
                "recent_legitimate_count": len(recent_legitimate),
                "fraud_signals": list(dict.fromkeys(fraud_signals))[:4],
                "legitimacy_signals": list(dict.fromkeys(legitimacy_signals))[:4],
                "uncertainties": uncertainties,
            }

    def get_behavior_baseline(
        self,
        user: UserProfile | None,
        transaction_time,
        transactions_data: list[TransactionRecord] | None = None,
        current_transaction: TransactionRecord | None = None,
    ) -> dict[str, Any]:
        with self.tracer.span(
            "get_behavior_baseline",
            as_type="tool",
            input_payload={
                "user": user.full_name if user else None,
                "transaction_id": current_transaction.transaction_id if current_transaction else None,
            },
        ):
            if user is None or current_transaction is None:
                return {
                    "history_count": 0,
                    "usual_hours": [],
                    "common_transaction_types": [],
                    "common_payment_methods": [],
                    "common_locations": [],
                    "common_counterparties": [],
                    "is_new_counterparty": False,
                    "is_new_payment_method": False,
                    "is_new_transaction_type": False,
                    "is_unusual_hour": False,
                    "is_unusual_amount": False,
                    "routine_match": False,
                }

            current_role = "sender" if current_transaction.sender_iban == user.iban else "recipient"
            history = [
                item
                for item in (transactions_data or self.dataset.user_transactions.get(user.iban, []))
                if item.timestamp < transaction_time
            ]
            amounts = sorted(item.amount for item in history)
            median_amount = statistics.median(amounts) if amounts else 0.0
            p90_index = max(0, int(0.9 * (len(amounts) - 1))) if amounts else 0
            p90_amount = amounts[p90_index] if amounts else 0.0

            type_counts = Counter(item.transaction_type for item in history)
            payment_counts = Counter(item.payment_method for item in history if item.payment_method)
            hour_counts = Counter(item.timestamp.hour for item in history)
            location_counts = Counter(filter(None, (transaction_city(item) for item in history)))
            counterparty_counts = Counter(
                transaction_counterparty_key(item, "sender" if item.sender_iban == user.iban else "recipient")
                for item in history
            )
            routine_keys = Counter(
                normalize_text(
                    merchant_hint(item.location, item.description, item.transaction_type)
                )
                for item in history
            )

            current_counterparty = transaction_counterparty_key(current_transaction, current_role)
            current_routine_key = normalize_text(
                merchant_hint(
                    current_transaction.location,
                    current_transaction.description,
                    current_transaction.transaction_type,
                )
            )

            unusual_amount = False
            if amounts:
                unusual_amount = current_transaction.amount > max(p90_amount * 1.8, median_amount * 2.4, 350.0)
            elif current_transaction.amount >= 700.0:
                unusual_amount = True

            top_hours = {hour for hour, _count in hour_counts.most_common(5)}
            top_types = [item for item, _count in type_counts.most_common(3)]
            top_methods = [item for item, _count in payment_counts.most_common(3)]
            top_locations = [item for item, _count in location_counts.most_common(4)]
            top_counterparties = [item for item, _count in counterparty_counts.most_common(5)]

            routine_match = False
            if counterparty_counts.get(current_counterparty, 0) >= 2:
                routine_match = True
            if routine_keys.get(current_routine_key, 0) >= 2:
                routine_match = True

            return {
                "history_count": len(history),
                "median_amount": round(median_amount, 2),
                "p90_amount": round(p90_amount, 2),
                "usual_hours": sorted(top_hours),
                "common_transaction_types": top_types,
                "common_payment_methods": top_methods,
                "common_locations": top_locations,
                "common_counterparties": top_counterparties,
                "is_new_counterparty": current_counterparty not in counterparty_counts,
                "is_new_payment_method": bool(current_transaction.payment_method)
                and current_transaction.payment_method not in payment_counts,
                "is_new_transaction_type": current_transaction.transaction_type not in type_counts,
                "is_unusual_hour": bool(hour_counts) and current_transaction.timestamp.hour not in top_hours,
                "is_unusual_amount": unusual_amount,
                "routine_match": routine_match,
            }

    def score_rule_based_risk(self, context_bundle: EvidenceBundle) -> dict[str, Any]:
        with self.tracer.span(
            "score_rule_based_risk",
            as_type="tool",
            input_payload={"transaction_id": context_bundle.transaction.transaction_id},
        ):
            risk = score_rule_based_risk(
                context_bundle,
                amount_high_impact_floor=self.config.amount_high_impact_floor,
                amount_high_impact_monthly_salary_factor=self.config.amount_high_impact_monthly_salary_factor,
            )
            return {
                "risk_score": risk.risk_score,
                "fraud_signals": risk.fraud_signals,
                "legitimacy_signals": risk.legitimacy_signals,
                "uncertainties": risk.uncertainties,
                "economic_high_impact": risk.economic_high_impact,
                "economic_amount_threshold": risk.economic_amount_threshold,
            }

    def build_transaction_evidence_bundle(
        self,
        transaction_id: str,
        dataset_path: str | None = None,
    ) -> EvidenceBundle:
        with self.tracer.span(
            "build_transaction_evidence_bundle",
            as_type="tool",
            input_payload={"transaction_id": transaction_id, "dataset_path": dataset_path},
        ):
            transaction = self.dataset.transactions_by_id[transaction_id]
            parties = resolve_transaction_parties(transaction, self.dataset.users_by_iban)
            focal_user, focal_role, counterparty_user = determine_focal_user(
                transaction,
                self.dataset.users_by_iban,
            )
            transaction_context = self.get_transaction_context(transaction_id, dataset_path)
            party_context = self.resolve_transaction_parties(transaction)
            location_context = self.get_user_location_context(
                focal_user,
                transaction.timestamp,
                self.dataset.locations_by_biotag,
                self.config.location_window_hours,
                transaction,
            )
            message_context = self.get_user_message_context(
                focal_user,
                transaction.timestamp,
                None,
                None,
                self.config.message_lookback_days,
                transaction,
            )
            behavior_baseline = self.get_behavior_baseline(
                focal_user,
                transaction.timestamp,
                self.dataset.user_transactions.get(focal_user.iban, []) if focal_user else None,
                transaction,
            )

            provisional = EvidenceBundle(
                transaction=transaction,
                focal_user=focal_user,
                counterparty_user=counterparty_user,
                focal_role=focal_role,
                transaction_context=transaction_context,
                party_context=party_context,
                location_context=location_context,
                message_context=message_context,
                behavior_baseline=behavior_baseline,
                risk_assessment=score_rule_based_risk(
                    EvidenceBundle(
                        transaction=transaction,
                        focal_user=focal_user,
                        counterparty_user=counterparty_user,
                        focal_role=focal_role,
                        transaction_context=transaction_context,
                        party_context=party_context,
                        location_context=location_context,
                        message_context=message_context,
                        behavior_baseline=behavior_baseline,
                        risk_assessment=None,
                    ),
                    amount_high_impact_floor=self.config.amount_high_impact_floor,
                    amount_high_impact_monthly_salary_factor=self.config.amount_high_impact_monthly_salary_factor,
                ),
            )
            return provisional