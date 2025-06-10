# Repository Guidelines

The instructions in this document apply to all files in this repository.

## Coding conventions

- Use 4 spaces for indentation in Python files.
- Limit lines to 100 characters when possible.
- Provide a docstring for any new function or class.
- Prefer snake_case for variable and function names.

## Testing

After modifying Python code, run the following command to ensure there are no syntax errors:

```bash
python -m py_compile <changed file(s)>
```

## Pull request notes

Include a short summary of the key changes and show the output of the syntax check in the testing section of the PR body.
