# Deployment Options

How to run an agent application in production. Each option trades off cost,
control, scalability, and operational effort.

## Local / on-premises (Ollama, vLLM)

- Run open-weight models on your own hardware.
- Pros: no per-token fees, full data privacy, works offline, predictable cost.
- Cons: you own the hardware, scaling, and uptime; upfront GPU cost.
- Best for: privacy-sensitive teams, steady high volume, air-gapped environments.

## Managed model APIs (OpenAI, Anthropic)

- Call a hosted model over HTTPS; no infrastructure to run.
- Pros: zero ops, best-in-class models, instant scaling.
- Cons: per-token cost, data leaves your network, vendor dependency.
- Best for: fast time-to-market and teams without ML infrastructure.

## Serverless containers (AWS Lambda, Google Cloud Run, Azure Container Apps)

- Package the app in a container that scales to zero when idle.
- Pros: pay only for usage, automatic scaling, low idle cost.
- Cons: cold starts, execution time limits, statelessness needs external memory.
- Best for: spiky or unpredictable traffic, cost-conscious production.

## Always-on containers / VMs (ECS, Kubernetes, a plain VM)

- A long-running service behind a load balancer.
- Pros: no cold starts, full control, good for websockets/streaming.
- Cons: you pay for idle capacity; more ops overhead.
- Best for: steady traffic and latency-sensitive workloads.

## Platform-as-a-Service (Streamlit Community Cloud, Hugging Face Spaces, Render, Fly.io)

- Push code and the platform handles hosting.
- Pros: fastest path to a public demo, minimal ops, generous free tiers.
- Cons: less control, resource caps, may sleep when idle.
- Best for: demos, internal tools, MVPs, hackathon projects.

## Decision guide

- Need a public demo today: Streamlit Community Cloud or Hugging Face Spaces.
- Bursty traffic, minimize cost: serverless containers (Cloud Run / Lambda).
- Steady traffic, need streaming: always-on container on ECS/Kubernetes.
- Strict privacy or offline: self-host open-weight models with Ollama or vLLM.
- Small team, no ops budget: managed model API + a PaaS front end.

## Cost drivers

- Model token pricing (or GPU rental for self-hosting) dominates most budgets.
- Idle compute on always-on hosts adds up; scale-to-zero avoids it.
- Egress, logging, and vector-database hosting are common hidden costs.
