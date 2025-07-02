---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Environment Information
Please complete the following information:
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12.0]
- Python Version: [e.g. 3.9.7]
- IRST Library Version: [e.g. 1.0.0]
- PyTorch Version: [e.g. 1.12.0]
- CUDA Version: [e.g. 11.6, or "CPU only"]
- GPU: [e.g. RTX 3080, or "None"]

## Code to Reproduce
```python
# Please provide minimal code that reproduces the issue
from irst_library import IRSTDetector

detector = IRSTDetector.from_pretrained("serank_sirst")
# ... rest of your code
```

## Error Messages/Logs
```
Paste any error messages or relevant log output here
```

## Screenshots
If applicable, add screenshots to help explain your problem.

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have an idea of what might be causing the issue or how to fix it, please describe it here.

## Checklist
- [ ] I have searched the existing issues to make sure this is not a duplicate
- [ ] I have provided all the required information above
- [ ] I have tested this with the latest version of IRST Library
- [ ] I have included minimal code to reproduce the issue
