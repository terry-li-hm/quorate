# Kimi subscription terms and route review

## Decision

Retain the Allegretto plan and K3 trial, but route every headless or scheduled Quorate call through Kimi's documented membership API. Keep the official CLI only for a human-run interactive terminal, and fail closed when the membership API key is unavailable. Do not expand the existing public, synthetic, and deliberately sanitized data boundary.

## Official evidence

The public international annual Allegretto price was US$372 on 17 July 2026. The actual Chinese price is the amount shown at checkout and may differ by region. The [Kimi Terms of Service](https://www.kimi.com/user/agreement/modelUse?version=v2), effective 21 January 2026, state that subscriptions renew automatically unless cancelled, website subscriptions may be cancelled through the product or `membership@moonshot.ai`, cancellation must occur at least 24 hours before renewal, paid fees are generally final and non-refundable, and subscription price increases require at least 30 days' notice.

The [Chinese paid-service agreement](https://www.kimi.com/user/agreement/paidServiceAgreement?version=v2), also effective 21 January 2026, controls the Chinese paid service where it differs from the general terms. It says automatic renewal applies when the member selected continuous monthly or annual renewal, cancellation can be performed through Kimi or the original WeChat, Alipay, or Apple payment channel, annual-plan benefits are issued monthly and expire rather than rolling over, and successful virtual-service purchases are generally non-refundable and non-transferable. It also confirms that the checkout price governs because regional prices may differ.

The same terms prohibit bots, scripts, and scheduled access that simulates human use unless Moonshot authorizes it. The [Kimi Code documentation](https://www.kimi.com/code/docs/en/) explicitly authorizes members to create API keys and integrate the service into third-party development tools, subject to preserving the tool's real identity. Quorate therefore prefers the documented API and identifies itself as `quorate`; an API key is required for non-interactive use.

The [membership documentation](https://www.kimi.com/code/docs/en/kimi-code/membership.html) says quota refreshes every seven days, unused quota does not roll over, all devices and API keys share quota and a rolling five-hour rate window, and a new coding-focused membership system is coming while active subscribers retain their combined benefits. K3 is available from Moderato upward, while Allegretto unlocks its one-million-token context window.

The [Privacy Policy](https://www.kimi.com/user/agreement/userPrivacy?version=v2) and Terms state that prompts and generated content may be processed to improve and train the service. The Terms provide an opt-out for model improvement and research through `membership@moonshot.ai`. Quorate's data boundary remains unchanged even after an opt-out.

## Account facts still requiring authentication

The public sources cannot establish the actual purchase price, purchase date, renewal date, payment channel, current remaining quota, or whether automatic renewal is enabled on this account. Those facts must be read from the authenticated subscription and Kimi Code Console pages. No cancellation, renewal, purchase, or Extra Usage setting was changed during this review.
