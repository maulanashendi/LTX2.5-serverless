# Frontend submission targets

The frontend accepts RunPod submission URLs matching
`https://api.runpod.ai/v2/<endpoint_id>/run` or `/runsync`.

To submit to a local API or another service, set `FRONTEND_SUBMIT_URLS` on the
frontend server to a comma-separated list of exact trusted POST URLs. For example:

```env
FRONTEND_SUBMIT_URLS=http://127.0.0.1:8000/runsync,https://worker.example.com/runsync
```

Only those exact URLs are accepted; paths, query strings, and fragments cannot
be changed by callers. Configure only services you trust to receive submitted
payloads and bearer tokens. Restart the frontend after changing the variable.
Submission requests do not follow redirects; enter the final destination URL.

Pod status polling always uses `LOCAL_COMFY_NODE`. The optional `node` query
parameter must match that configured node. The frontend already sends the node
returned by pod submission.
