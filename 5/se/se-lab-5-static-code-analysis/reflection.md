
# SE Lab-5: Static Code Analysis

## 1. Which issues were the easiest to fix, and which were the hardest? Why?

**Easiest to fix:**

- **Removing the eval() call** was the simplest fix since it could be removed, while retaining the print, without affecting functionality.
- **Adding the missing final newline** required just adding a blank line at the end of the file.
- **Removing unused import** was straightforward, by deleting the `import logging` statement.
- **Adding encoding parameters** was simple, by just adding `encoding='utf-8'` to both `open()` calls.

**Hardest to fix:**

- **Mutable default argument** required understanding knowledge of how Python handles default arguments. The fix involved changing the signature to `logs=None` and adding conditional initialization inside the function.
- **Refactoring to use context managers** required restructuring the file operations with `with` statements, which affected the code flow and required careful indentation management.
- **Fixing the bare except clause** required understanding what specific exceptions could occur during dictionary deletion. Choosing the right exception type (KeyError) needed consideration of the actual error scenarios.
- **Function renaming (multiple lines)** was tedious because all function calls throughout the code also needed to be updated to maintain consistency.

## 2. Did the static analysis tools report any false positives? If so, describe one example.

The global statement warning could be considered a borderline false positive in this specific context. While Pylint flags the use of `global stock_data` as poor practice, in a small script like this inventory system, using a global variable for shared state is a reasonable design choice. Refactoring to eliminate the global would require significant architectural changes (like creating a class or passing the dictionary as a parameter), which may be overkill for a simple script. This isn't a true false positive, as it's a legitimate issue that Pylint correctly identified. The tool is warning about potential maintainability issues, even if the code works correctly.

## 3. How would you integrate static analysis tools into your actual software development workflow? Consider continuous integration (CI) or local development practices.

**Local Development Practices:**
- **Pre-commit hooks**: Set up Git hooks that automatically run Flake8 and Pylint before allowing commits, ensuring code quality standards are met before code enters version control.
- **IDE integration**: Configure VS Code, PyCharm, or other IDEs to run these tools in real-time, showing warnings and errors as you type with inline annotations.
- **Make targets**: Create a `Makefile` with targets like `make lint` that runs all three tools sequentially for quick local validation.

**Continuous Integration (CI) Pipeline:**
- **Automated checks on pull requests**: Configure GitHub Actions, GitLab CI, or Jenkins to run Pylint, Flake8, and Bandit on every PR, blocking merges if critical issues are found.
- **Quality gates**: Set minimum code quality scores (e.g., Pylint score ≥ 8.0/10) that must be met for builds to pass.
- **Security scanning**: Run Bandit specifically in CI to catch security vulnerabilities before they reach production, with high/medium severity issues causing build failures.
- **Reporting and trends**: Generate reports showing code quality trends over time, helping teams track improvements.

**Practical Workflow:**
1. Run tools locally during development
2. Fix issues before committing
3. CI validates on push/PR
4. Code review includes automated tool feedback
5. Merge only after passing all checks

## 4. What tangible improvements did you observe in the code quality, readability, or potential robustness after applying the fixes?

**Security Improvements:**
- Removing `eval()` eliminated arbitrary code execution vulnerability, making the code significantly safer.
- Adding explicit encoding to file operations prevents encoding-related bugs across different systems and locales.
- Replacing bare `except:` with specific exception handling makes error handling more predictable and prevents masking unexpected errors.

**Bug Prevention:**
- Fixing the mutable default argument prevents a subtle but serious bug where the logs list would be shared across function calls, causing unexpected state persistence.
- Using context managers (`with` statements) ensures files are properly closed even if exceptions occur, preventing resource leaks.

**Readability Improvements:**
- Converting to f-strings makes string formatting more readable and modern.
- Following snake_case naming conventions makes the code consistent with Python community standards and easier to read for other developers.
- Proper PEP 8 spacing with 2 blank lines between functions improves visual organization and makes the code structure clearer.

**Maintainability:**
- Adding docstrings provides clear documentation of what each function does, making the codebase easier to understand and maintain.
- Removing unused imports reduces clutter and makes dependencies clearer.

---

