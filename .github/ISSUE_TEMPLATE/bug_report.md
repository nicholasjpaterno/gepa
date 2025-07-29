---
name: 🐛 Bug Report
about: Report a bug to help us improve GEPA
title: "[BUG] "
labels: bug, needs-triage
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

Please provide detailed steps to reproduce the issue:

1. Go to '...'
2. Run command '...'
3. Set configuration '...'
4. See error

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

A clear and concise description of what actually happened.

## 📋 Environment

Please complete the following information:

- **GEPA Version**: [e.g., 0.1.0]
- **Python Version**: [e.g., 3.11.0]
- **Operating System**: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- **LLM Provider**: [e.g., OpenAI, Anthropic, Ollama]
- **Model**: [e.g., gpt-4, claude-3-opus, llama2]

## 📝 Configuration

Please share your GEPA configuration (remove any sensitive information like API keys):

```yaml
# Your config.yaml or relevant configuration
inference:
  provider: "openai"
  model: "gpt-4"
  # ... other settings

optimization:
  budget: 50
  # ... other settings
```

## 🔧 Code Sample

Please provide a minimal code sample that reproduces the issue:

```python
import asyncio
from gepa import GEPAOptimizer, GEPAConfig

# Your minimal reproduction code here
async def main():
    config = GEPAConfig(...)
    # ... rest of the code that causes the issue

asyncio.run(main())
```

## 📊 Error Output

If applicable, paste the full error message and stack trace:

```
Traceback (most recent call last):
  File "...", line ..., in ...
    ...
Error: ...
```

## 📸 Screenshots

If applicable, add screenshots to help explain your problem.

## 💡 Possible Solution

If you have ideas on how to fix the issue, please describe them here.

## 📚 Additional Context

Add any other context about the problem here:

- Does this happen consistently or intermittently?
- Are you using any specific datasets or custom metrics?
- Any recent changes to your setup?
- Related issues or discussions?

## ✅ Checklist

Please check the following before submitting:

- [ ] I have searched existing issues and this is not a duplicate
- [ ] I have provided all the requested information above
- [ ] I have removed any sensitive information (API keys, personal data)
- [ ] I have tested with the latest version of GEPA
- [ ] I have included a minimal reproduction case

## 🏷️ Priority

How critical is this issue for your use case?

- [ ] 🔴 Critical - Blocking production use
- [ ] 🟡 High - Significant impact on functionality
- [ ] 🟢 Medium - Some impact but workarounds exist
- [ ] 🔵 Low - Minor issue or enhancement

---

**Thank you for taking the time to report this issue! Your feedback helps make GEPA better for everyone.** 🚀