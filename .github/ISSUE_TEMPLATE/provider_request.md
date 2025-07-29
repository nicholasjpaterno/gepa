---
name: 🔌 Provider Request
about: Request support for a new LLM provider or improve existing provider integration
title: "[PROVIDER] Add support for "
labels: provider, enhancement
assignees: ''
---

## 🔌 Provider Information

**Which LLM provider would you like to see supported?**

- **Provider Name**: [e.g., Cohere, Together AI, HuggingFace Inference]
- **Provider Website**: [e.g., https://cohere.com]
- **API Documentation**: [Link to API docs]
- **Provider Type**: 
  - [ ] ☁️ Cloud API Service
  - [ ] 🏠 Self-hosted Solution
  - [ ] 🔗 Model Hub/Marketplace
  - [ ] 🛠️ Local Inference Engine

## 🎯 Use Case

**Why do you need this provider?**

Describe your specific use case and why existing providers don't meet your needs:

- Model capabilities you need
- Cost considerations
- Performance requirements
- Privacy/security requirements
- Regional availability
- Specific model access

## 📋 Provider Details

### Models Available
List the models you'd like to use:

- [ ] **Model 1**: [e.g., command-xlarge-nightly] - [Description]
- [ ] **Model 2**: [e.g., command-medium] - [Description]
- [ ] **Model 3**: [e.g., command-light] - [Description]

### API Features
Which API features does this provider support?

- [ ] 💬 Text Completion
- [ ] 🗨️ Chat Completion
- [ ] 🌊 Streaming Responses
- [ ] 🔢 Token Counting
- [ ] 💰 Usage/Cost Tracking
- [ ] ⚙️ Temperature Control
- [ ] 🎯 Max Tokens Setting
- [ ] 🚫 Stop Sequences
- [ ] 🔄 Batch Processing

### Authentication
How does authentication work?

- [ ] 🔑 API Key
- [ ] 🎫 Bearer Token
- [ ] 📜 OAuth
- [ ] 🆔 Username/Password
- [ ] 🔒 Custom Headers
- [ ] 📋 Other: [Describe]

## 💻 API Example

**If you have experience with this provider, show a basic API call:**

```python
# Example API usage (remove any real API keys)
import requests

response = requests.post(
    "https://api.provider.com/v1/generate",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "model": "model-name",
        "prompt": "Your prompt here",
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print(response.json())
```

## 🔍 Research Done

**What research have you done?**

- [ ] 📖 Read the provider's API documentation
- [ ] 💻 Tested the provider's API directly
- [ ] 🔍 Looked for existing Python clients/SDKs
- [ ] 📊 Compared with other providers
- [ ] 💬 Asked in provider's community/support

**Relevant Links:**
- API Documentation: [URL]
- Python SDK: [URL if available]
- Pricing Information: [URL]
- Model Documentation: [URL]

## 🛠️ Implementation Considerations

### Challenges
What challenges might there be in implementing this provider?

- [ ] 🔄 Non-standard API format
- [ ] 📊 Limited model information
- [ ] 💰 Complex pricing structure
- [ ] 🔒 Special authentication requirements
- [ ] 📈 Rate limiting considerations
- [ ] 🌍 Regional availability issues
- [ ] 📋 Other: [Describe]

### Dependencies
Would this require any new dependencies?

- [ ] No new dependencies needed
- [ ] Provider's official SDK: [Name and version]
- [ ] HTTP client library: [Name and version]
- [ ] Authentication library: [Name and version]  
- [ ] Other: [Describe]

## 📊 Priority & Impact

**How important is this provider for your use case?**

- [ ] 🔴 Critical - Required for production use
- [ ] 🟡 High - Would significantly improve our workflow
- [ ] 🟢 Medium - Nice to have for experimentation
- [ ] 🔵 Low - Interested but not urgent

**Potential Impact:**
- [ ] 🌟 Opens new market/user segment
- [ ] 💰 Provides cost-effective alternative
- [ ] 🚀 Better performance for specific tasks
- [ ] 🔒 Better privacy/security compliance
- [ ] 🌍 Better regional availability
- [ ] 🎯 Access to specialized models

## 🤝 Community Support

**Is there community interest in this provider?**

- [ ] Multiple users have requested this
- [ ] Discussed in Discord/community channels
- [ ] Related to common use cases
- [ ] Part of popular provider ecosystem

## 💪 Contribution

**Can you help with the implementation?**

- [ ] 💻 I can implement the provider client
- [ ] 🧪 I can help with testing
- [ ] 📚 I can help with documentation
- [ ] 🔍 I can help with research
- [ ] 💰 I can sponsor development
- [ ] 🤔 I'd prefer someone else implements it

**If you can contribute, what's your experience level?**
- [ ] 🆕 New to GEPA internals
- [ ] 📚 Familiar with GEPA architecture
- [ ] 🏆 Experienced with provider integrations
- [ ] 🔬 Expert in LLM APIs

## 📋 Additional Information

**Anything else we should know?**

- Provider-specific quirks or limitations
- Best practices for using this provider
- Community resources or examples
- Alternative providers you've considered

## ✅ Checklist

Please check the following before submitting:

- [ ] I have searched existing issues for this provider
- [ ] I have provided comprehensive provider information
- [ ] I have included API documentation links
- [ ] I have described my specific use case
- [ ] I understand this is a community-driven project

---

**Thank you for suggesting this provider! Provider diversity makes GEPA more powerful for everyone.** 🚀

## 🎯 What Happens Next?

1. **Community Discussion**: Others can express interest and provide input
2. **Technical Review**: Maintainers will assess implementation complexity
3. **Implementation**: Community members or maintainers implement the provider
4. **Testing**: Thorough testing with real API credentials
5. **Documentation**: Examples and integration guides
6. **Release**: Included in the next GEPA version

Want to track progress? Watch this issue for updates! 👀