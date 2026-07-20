# Haikyuu RAG TXT Format

Put `.txt` files in this folder, then call `POST /reindex`.

Nested folders are supported. For character-specific files, use:

```text
characters/<character>/<character>_<category>.txt
```

Empty `.txt` files are skipped by the indexer until you add body text.

Files can be plain text, or they can start with an optional YAML metadata header:

```txt
---
title: Karasuno vs Aoba Johsai
type: match_summary
season: 2
episode: 24
characters:
  - Hinata
  - Kageyama
  - Oikawa
---

Main document text here.
```

Only `.txt` files are indexed.
