# Security Policy

## Supported Versions

We provide security updates for the following versions of GEPA:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously and encourage responsible disclosure of security vulnerabilities.

### How to Report

1. **GitHub Security Advisories** (Preferred): 
   - Go to the [Security Advisories](https://github.com/nicholasjpaterno/gepa/security/advisories) page
   - Click "Report a vulnerability"
   - Provide detailed information about the vulnerability

2. **Private Issue**: 
   - Create a GitHub issue with the `security` label
   - Mark it as private if possible
   - Include detailed reproduction steps

### What to Include

Please include as much of the following information as possible:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Assessment**: Within 1 week
- **Fix Development**: Depends on complexity and severity
- **Public Disclosure**: After fix is released (coordinated disclosure)

## Security Best Practices

### For Users

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use `.env` files and add them to `.gitignore`
3. **Network**: Use HTTPS for all API endpoints
4. **Dependencies**: Keep dependencies updated
5. **Secrets Management**: Use proper secrets management solutions in production

### For Contributors

1. **Input Validation**: Always validate user inputs
2. **SQL Injection**: Use parameterized queries
3. **XSS Prevention**: Sanitize outputs
4. **Authentication**: Implement proper authentication and authorization
5. **Error Handling**: Don't expose sensitive information in error messages

## Known Security Considerations

### LLM API Interactions
- API keys are transmitted over HTTPS
- No user data is stored permanently without explicit consent
- Prompt data may be sent to third-party LLM providers based on configuration

### Database Security
- Uses SQLAlchemy ORM to prevent SQL injection
- Database credentials should be stored securely
- Supports encrypted connections to PostgreSQL

### Local Model Security
- Local inference servers (Ollama, LMStudio) run on localhost by default
- Network access should be restricted in production environments
- Model files should come from trusted sources

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all supported versions
4. Release new versions as soon as possible
5. Credit the reporter (unless they prefer to remain anonymous)

## Contact

For security-related questions or concerns, please use the reporting methods above rather than public channels.

## License

This security policy is licensed under [MIT License](LICENSE).