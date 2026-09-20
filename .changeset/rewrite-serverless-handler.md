---
"ltx25-worker": patch
---

Rewrite the serverless handler to a strict RunPod handler contract: exceptions now propagate instead of being swallowed into `{"status": "error"}`, ComfyUI readiness is checked via `/system_stats` before a job renders (fixing cold-start failures), and a new flat input path (`{"prompt", "image"?}`) picks text-to-video or image-to-video automatically from whether an image is supplied. The raw `workflow` input path is unchanged. Removed the legacy fleet failover, circuit breaker, VIP rate limiter, and API-key check; Redis stays for local caching/dedup but every call is now best-effort and the dedup wait is capped at 60 seconds instead of blocking for 20 minutes.
