## Cline Rule: Tombstone Emoji Usage in Codebase Print Statements

### Rule: **NEVER use emojis in print statements or console output**

### Rationale:

**1. Cross-Platform Compatibility Issues**
- Emojis cause `UnicodeEncodeError` on Windows systems with default console encoding (cp1252)
- Different operating systems handle Unicode differently, leading to inconsistent behavior
- CI/CD pipelines often run on Windows agents where emoji encoding fails

**2. Terminal and Logging Compatibility**
- Many terminal emulators don't support full Unicode emoji rendering
- Log files and output redirection may corrupt emoji characters
- Automated tools parsing output may fail on unexpected Unicode characters

**3. Accessibility and Readability**
- Screen readers may not properly handle emoji characters
- Some users have limited Unicode support in their environments
- Plain text is universally readable and searchable

**4. Performance and Bundle Size**
- Unicode characters increase string processing overhead
- Emojis can cause encoding/decoding performance issues
- Larger character sets may impact memory usage

**5. Maintenance and Debugging**
- Emojis make log parsing and automated monitoring more difficult
- Debugging encoding issues is time-consuming and error-prone
- Plain ASCII text is more reliable for system diagnostics

### Recommended Alternatives:
- Use descriptive text: `[OK]`, `[FAIL]`, `[ERROR]`, `[SUCCESS]`
- Use structured logging with consistent prefixes
- Implement color coding through terminal escape sequences instead of Unicode symbols

### Implementation:
- Replace all emoji characters in existing code with plain text equivalents
- Add linting rules to prevent future emoji usage in print statements
- Use consistent status indicators across the entire codebase
- Test output on multiple platforms including Windows, Linux, and macOS

This rule ensures maximum compatibility, accessibility, and maintainability of the codebase across all deployment environments.