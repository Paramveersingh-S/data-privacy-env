# 📈 Scaling & Complexity

`DataPrivacyEnv` is designed to scale across three dimensions:

## 1. Data Volume
While the current environment uses a small mock dataset, the `_get_initial_state` method can be extended to pull from **synthetic data generators**. This allows for "Needle in a Haystack" testing where an agent must scan millions of logs to find a single leak.

## 2. Task Complexity
Future iterations include **Multi-Step Dependency** tasks:
- **Contextual Redaction**: Redacting a name only if it appears near a sensitive keyword (e.g., "Salary").
- **Cross-Endpoint Verification**: Deleting a user in the CRM and then verifying their removal from the Auth-Server.

## 3. Compliance Diversity
The environment can be easily tuned for different regulatory frameworks:
- **GDPR**: Focus on "Right to be Forgotten" (Deletion).
- **PCI-DSS**: Focus on Credit Card Number (CCN) masking in transaction logs.
- **HIPAA**: Focus on Patient Health Information (PHI) anonymization.
