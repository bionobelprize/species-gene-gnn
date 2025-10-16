# Contributing to Species-Gene GNN

Thank you for your interest in contributing to the Species-Gene GNN project!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Enhancements

We welcome feature requests! Please open an issue with:
- A clear description of the enhancement
- Use cases and examples
- Any potential implementation ideas

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -am 'Add some feature'`)
7. Push to the branch (`git push origin feature/your-feature-name`)
8. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/species-gene-gnn.git
cd species-gene-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Testing

- Write tests for all new features
- Ensure existing tests pass
- Aim for high code coverage

### Documentation

- Update README.md if you add new features
- Add docstrings to all new functions/classes
- Update examples if necessary

## Code of Conduct

Be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## Questions?

Feel free to open an issue for any questions or clarifications!
