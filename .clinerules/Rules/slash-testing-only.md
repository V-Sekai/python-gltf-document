# Cline Rule: Slash Framework Testing Only

## Rule: **MANDATORY SLASH FRAMEWORK FOR ALL TESTS**

### ✅ REQUIRED Testing Framework:

- **Slash Framework Only** - All tests must use slash framework exclusively
- No other testing frameworks allowed (pytest, unittest, nose, etc.)
- All test files must follow slash conventions and patterns

### ❌ FORBIDDEN Testing Frameworks:

- pytest (pytest.mark, pytest.fixture, etc.)
- unittest (TestCase classes, setUp/tearDown methods)
- nose (nose-specific decorators and patterns)
- doctest (embedded docstring tests)
- Any other testing framework not based on slash

### 📝 Slash Framework Requirements:

**Test File Structure:**

```python
# test_example.py
import slash

@slash.fixture
def gltf_document():
    """Fixture for GLTF document testing"""
    return GLTFDocument()

@slash.test
def test_gltf_loading(gltf_document):
    """Test GLTF file loading functionality"""
    # Test implementation using slash
    pass

@slash.test
def test_gltf_export(gltf_document):
    """Test GLTF export functionality"""
    # Test implementation using slash
    pass
```

**Test Organization:**

- Use `@slash.test` decorator for test functions
- Use `@slash.fixture` for test fixtures and setup
- Use `@slash.parametrize` for parameterized tests
- Use `@slash.skip` and `@slash.tag` for test control

**Test Discovery:**

- All test files must be named `test_*.py` or `*_test.py`
- Tests must be placed in `tests/` directory
- Use slash's automatic test discovery

### 🎯 Rationale:

**1. Consistency**

- Single testing framework across the entire codebase
- Standardized test patterns and conventions
- Predictable test behavior and output

**2. Features**

- Slash provides advanced testing features
- Better test organization and reporting
- Improved debugging and failure analysis
- Enhanced test parallelization

**3. Maintenance**

- Single framework to maintain and update
- Consistent documentation and examples
- Easier onboarding for new developers
- Reduced complexity in CI/CD pipelines

**4. Integration**

- Seamless integration with existing codebase
- Compatible with current development workflow
- Supports all required testing scenarios

### 🔧 Implementation:

**Pre-commit Hooks:**

- Validate that test files use slash framework only
- Reject imports of forbidden testing frameworks
- Check for slash-specific patterns and decorators

**CI/CD Integration:**

- Run tests using slash runner exclusively
- Generate reports in slash format
- Validate test coverage with slash tools

**Code Review:**

- Reviewers must verify slash framework usage
- Flag any usage of forbidden testing frameworks
- Ensure proper slash patterns and conventions

### ⚠️ Migration Requirements:

**Existing Test Files:**

- Convert all existing test files to use slash framework
- Remove imports of pytest, unittest, and other frameworks
- Update test patterns to use slash decorators
- Maintain test coverage and functionality

**Examples:**

```python
# ❌ FORBIDDEN - pytest
import pytest

@pytest.mark.parametrize("format", ["gltf", "glb"])
def test_formats(format):
    pass

# ✅ REQUIRED - slash
import slash

@slash.parametrize("format", ["gltf", "glb"])
def test_formats(format):
    pass
```

### 📋 Enforcement:

- Pre-commit hooks reject non-slash test files
- CI/CD fails on forbidden framework usage
- Code reviews flag non-compliant tests
- Automated tools validate framework usage

This rule ensures all tests use the slash framework exclusively, providing consistency, advanced features, and better maintainability across the entire codebase.
