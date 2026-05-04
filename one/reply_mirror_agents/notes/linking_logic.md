# Linking Logic

## Dataset paths actually present

- `one/The Truman Show - train`
- `one/The Truman Show - validation`

## Robust joins

- `transactions.sender_iban` and `transactions.recipient_iban` join to `users.iban`.
- `locations.biotag` maps to a user via the city code embedded in the biotag and the user's residence city.
- For the current splits, each user has a unique residence city, so city-code matching is stable.

## Transaction interpretation

- The focal user is the internal user whose IBAN appears on the transaction.
- The focal role is `sender` for outgoing user payments and `recipient` for incoming credits such as salary.
- The counterparty key is derived from the opposite IBAN or ID, falling back to merchant location or description.

## Message linkage

- `sms.json` stores raw SMS threads under `sms`.
- `mails.json` stores raw email threads under `mail`.
- Messages are linked to users using full name, first name, recipient header, and email-local-part style matching.
- Mail parsing uses standard headers plus stripped HTML body text.

## Fraud signals emphasized

- New counterparty or merchant
- Amount far above historical norms
- New payment method or transaction type
- In-person or withdrawal location inconsistent with nearby GPS observations
- Recent phishing or suspicious-login messages
- High-value suspicious cases receive stronger escalation

## Legitimacy signals emphasized

- Recurring salary, rent, bill, or other routine patterns
- Prior history with the same counterparty
- Legitimate order or billing communication near the transaction time
- GPS evidence consistent with in-person transaction city