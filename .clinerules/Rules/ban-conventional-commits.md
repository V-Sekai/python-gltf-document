# Cline Rule: Ban Conventional Commits

## Rule: **CONVENTIONAL COMMITS ARE BANNED**

### ❌ FORBIDDEN Commit Formats:

- `feat:` `fix:` `docs:` `style:` `refactor:` `test:` `chore:` prefixes
- `(scope)` parentheses in commit messages
- Any structured commit format
- Conventional commits specification compliance
- Type prefixes of any kind

### ✅ REQUIRED Commit Style:

- Plain English descriptions only
- Clear, descriptive messages
- No prefixes or structured formats
- Direct communication style
- Human-readable commit messages

### 📝 Examples:

**❌ WRONG (Banned):**

```
feat(gltf): add new accessor decoder
fix: resolve buffer parsing issue
docs: update README with installation instructions
style: format code with black
refactor: simplify GLTF state management
test: add unit tests for buffer validation
chore: update dependencies
```

**✅ CORRECT (Required):**

```
Add new accessor decoder for GLTF buffers
Fix buffer parsing issue in GLTF loader
Update README with installation instructions
Format code with black
Simplify GLTF state management
Add unit tests for buffer validation
Update project dependencies
```

### 🎯 Rationale:

**1. Simplicity Over Structure**

- Plain English is universally understood
- No need to learn commit format conventions
- Direct communication reduces cognitive overhead

**2. Readability**

- Commit messages are for humans, not machines
- Clear descriptions improve code review process
- No parsing required to understand changes

**3. Flexibility**

- No rigid format restrictions
- Developers can write naturally
- Adapts to different project needs

**4. Tool Compatibility**

- Works with all Git tools and interfaces
- No special parsing required
- Compatible with existing workflows

### 🔧 Implementation:

- Add pre-commit hooks to reject conventional commit format
- Include in CI/CD pipeline validation
- Document in contributing guidelines
- Provide examples in commit message templates

### ⚠️ Enforcement:

- Pre-commit hooks will reject commits with conventional format
- CI/CD will fail on conventional commit messages
- Code reviews will flag non-compliant messages
- Automated tools will validate commit history

This rule ensures all commit messages are human-readable, clear, and follow plain English communication standards.
