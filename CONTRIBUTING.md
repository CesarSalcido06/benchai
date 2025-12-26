# Contributing to BenchAI

Thank you for your interest in contributing to BenchAI! This document provides guidelines and instructions for contributing.

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment

---

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear title describing the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, GPU, RAM)
   - Relevant logs

### Feature Requests

1. Check existing issues/discussions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Proposed implementation (optional)

### Pull Requests

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Test thoroughly
5. Commit with clear messages:
   ```bash
   git commit -m "Add: description of changes"
   ```
6. Push and create PR:
   ```bash
   git push origin feature/your-feature-name
   ```

---

## Development Setup

### Prerequisites

```bash
# Python 3.10+
python3 --version

# llama.cpp with CUDA
~/llama.cpp/build/bin/llama-cli --version

# ChromaDB
pip install chromadb

# FastAPI
pip install fastapi uvicorn
```

### Running Locally

```bash
cd router
python3 llm_router.py
```

### Testing

```bash
# Health check
curl http://localhost:8085/health

# Chat test
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general","messages":[{"role":"user","content":"ping"}],"max_tokens":10}'

# Metrics
curl http://localhost:8085/v1/metrics
```

---

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Keep functions focused and small

### Commits

- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues when applicable

### Documentation

- Update relevant docs with changes
- Add examples for new features
- Keep README current

---

## Architecture Guidelines

### Adding New Models

1. Add configuration to `MODELS` dict
2. Follow existing naming conventions
3. Document resource requirements
4. Test with target hardware

### Adding New Tools

1. Define async tool function
2. Add to `TOOLS` dictionary with schema
3. Implement proper error handling
4. Add tests and documentation

### Adding New Endpoints

1. Use FastAPI decorators
2. Include proper typing
3. Add to API documentation
4. Include error handling

---

## Release Process

1. Update `CHANGELOG.md`
2. Bump version in `README.md`
3. Create release tag
4. Update documentation

---

## Questions?

- Open a GitHub issue
- Check existing documentation
- Review closed issues for similar questions

---

*Thank you for contributing to BenchAI!*
