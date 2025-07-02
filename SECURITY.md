# Security Policy

## Supported Versions

We actively support the following versions of IRST Library with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.5.x   | :white_check_mark: |
| 1.0.x   | :warning: Limited  |
| < 1.0   | :x:                |

## Reporting a Vulnerability

### How to Report

If you discover a security vulnerability in IRST Library, please report it responsibly:

1. **Email**: Send details to <security@irst-library.org>
2. **Subject**: Include "SECURITY" in the subject line
3. **Details**: Provide a clear description of the vulnerability
4. **Impact**: Explain the potential impact and attack scenarios
5. **Reproduction**: Include steps to reproduce the issue
6. **Environment**: Specify affected versions and configurations

### What to Include

When reporting a security issue, please provide:

- **Description**: Clear description of the vulnerability
- **Steps to reproduce**: Detailed reproduction steps
- **Impact assessment**: Potential security implications
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have ideas for mitigation
- **Your contact information**: For follow-up questions

### Response Timeline

We are committed to addressing security issues promptly:

- **Initial Response**: Within 24 hours
- **Assessment**: Within 72 hours
- **Fix Development**: Within 1-2 weeks (depending on severity)
- **Public Disclosure**: After fix is available

### Severity Classification

We classify security vulnerabilities using the following criteria:

#### Critical (CVSS 9.0-10.0)

- Remote code execution
- Complete system compromise
- Unauthorized access to sensitive data

#### High (CVSS 7.0-8.9)

- Privilege escalation
- Significant data exposure
- Authentication bypass

#### Medium (CVSS 4.0-6.9)

- Limited data exposure
- Denial of service
- Information disclosure

#### Low (CVSS 0.1-3.9)

- Minor information leakage
- Limited impact vulnerabilities

## Security Measures

### Code Security

- **Static Analysis**: Automated security scanning with bandit
- **Dependency Scanning**: Regular vulnerability scans with safety
- **Code Review**: Mandatory security reviews for sensitive changes
- **Input Validation**: Comprehensive input sanitization

### Model Security

- **Model Integrity**: Cryptographic verification of pretrained models
- **Checksum Validation**: SHA-256 checksums for all model files
- **Source Verification**: Models signed by trusted sources
- **Malware Scanning**: Automated scanning of model files

### Infrastructure Security

- **Container Security**: Regular base image updates
- **Secret Management**: Secure handling of API keys and credentials
- **Network Security**: Encrypted communications
- **Access Control**: Role-based access control for repositories

### Runtime Security

- **Input Sanitization**: Robust validation of user inputs
- **Error Handling**: Secure error messages without information leakage
- **Logging**: Comprehensive audit logging
- **Monitoring**: Real-time security monitoring

## Best Practices for Users

### Secure Installation

```bash
# Verify package integrity
pip install irst-library --trusted-host pypi.org
pip check irst-library

# Verify model checksums
irst-verify --model-path model.pth --checksum sha256:abc123...
```

### Secure Configuration

```python
# Use secure defaults
from irst_library.security import SecureConfig

config = SecureConfig(
    validate_inputs=True,
    enable_logging=True,
    sandbox_mode=True
)
```

### Production Deployment

- **Use official Docker images**: Always use signed, official containers
- **Regular updates**: Keep the library and dependencies updated
- **Network isolation**: Deploy in isolated network environments
- **Monitoring**: Implement comprehensive logging and monitoring
- **Access control**: Implement proper authentication and authorization

## Security Updates

### Automatic Updates

- Critical security fixes are released immediately
- Security patches are backported to supported versions
- Users are notified through multiple channels

### Notification Channels

- **GitHub Security Advisories**: Primary notification method
- **Mailing List**: <security-announce@irst-library.org>
- **Documentation**: Updates posted to security documentation
- **Social Media**: Announcements on official accounts

## Responsible Disclosure

We follow responsible disclosure practices:

1. **Private reporting**: Initial vulnerability reports are kept confidential
2. **Coordinated disclosure**: We work with reporters to understand and fix issues
3. **Public disclosure**: Vulnerabilities are disclosed after fixes are available
4. **Credit**: Security researchers receive appropriate credit
5. **Hall of Fame**: Recognition for significant security contributions

## Security Hall of Fame

We recognize security researchers who help improve IRST Library security:

- [Your Name Here] - [Vulnerability Type] - [Date]
- [Future contributors]

## Contact Information

- **Security Team**: <security@irst-library.org>
- **General Issues**: <https://github.com/sachin-deshik-10/irst-library/issues>
- **Discussions**: <https://github.com/sachin-deshik-10/irst-library/discussions>

## Legal

- **Safe Harbor**: We will not pursue legal action against security researchers who follow responsible disclosure
- **Scope**: This policy applies to the IRST Library codebase and infrastructure
- **Exclusions**: Social engineering, physical attacks, and DoS attacks are out of scope

---

**Last Updated**: July 2, 2025
**Version**: 1.0
