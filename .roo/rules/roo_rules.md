---
description: Guidelines for creating and maintaining Roo Code rules to ensure consistency and effectiveness.
globs: .roo/rules/*.md
alwaysApply: true
---

- **Required Rule Structure:**
  ```markdown
  ---
  description: Clear, one-line description of what the rule enforces
  globs: path/to/files/*.ext, other/path/**/*
  alwaysApply: boolean
  ---

  - **Main Points in Bold**
    - Sub-points with details
    - Examples and explanations
  ```

- **File References:**
  - Use `[filename](mdc:path/to/file)` ([filename](mdc:filename)) to reference files
  - Example: [prisma.md](mdc:.roo/rules/prisma.md) for rule references
  - Example: [schema.prisma](mdc:prisma/schema.prisma) for code references

- **Code Examples:**
  - Use language-specific code blocks
  ```typescript
  // ✅ DO: Show good examples
  const goodExample = true;
  
  // ❌ DON'T: Show anti-patterns
  const badExample = false;
  ```

- **Rule Content Guidelines:**
  - Start with high-level overview
  - Include specific, actionable requirements
  - Show examples of correct implementation
  - Reference existing code when possible
  - Keep rules DRY by referencing other rules

- **Rule Maintenance:**
  - Update rules when new patterns emerge
  - Add examples from actual codebase
  - Remove outdated patterns
  - Cross-reference related rules

- **Best Practices:**
  - Use bullet points for clarity
  - Keep descriptions concise
  - Include both DO and DON'T examples
  - Reference actual code over theoretical examples
  - Use consistent formatting across rules 
  - Prioritise clean code, 'less is more'.
  - Use uv package to install any package - 'uv pip install...'

- **Vectorbtpro Library:**
  - When working within a specific task, prioritise rewrite of an existing script, especially when related to a particular subtask. Only create new scripts when necessary, to improve code organisation, modularity and if the script is required for independent functions.
  - Prioritise the use of the 'vectorbtpro' library methods, classes and functions.
  - Only use 'CCXT' or other exchange accessor libraries, for tasks requiring fetching of metadata for symbols.
  - All data objects should be created and accessible with the vbt.data oject methods available in vectorbtpro. We should not be relying on pandas.
  - Before attempting to implement something within vectorbtpro, ALWAYS check the terminal for what commands are available, and refer to the vectorbtpro documentation [api_data_custom_ccxt.md](mdc:vectorbtpro_docs/api_data_custom_ccxt.md) [cookbook_data.md](mdc:vectorbtpro_docs/cookbook_data.md) [api_data_custom_hdf.md](mdc:vectorbtpro_docs/api_data_custom_hdf.md) [features_data.md](mdc:vectorbtpro_docs/features_data.md) [documentation_data.md](mdc:vectorbtpro_docs/documentation_data.md) [cookbook_indexing.md](mdc:vectorbtpro_docs/cookbook_indexing.md) [cookbook_datetime.md](mdc:vectorbtpro_docs/cookbook_datetime.md)