# AGENTS.md
- Ты находишься в Windows.
- В репозитории есть virtual environment `subs_env`, перед запуском скриптов всегда активируй его.
- Скрипты запускай из корня репозитория через `uv` в виде модулей, например: `uv run -m utils.srt_to_json ...`.
- Для установки или удаления пакетов используй `uv pip install <package>` или `uv pip uninstall <package>`, но только после активации `subs_env`.

## Скрипты
- Всегда типизируй код.
- Тестируй код, после того как написал, чтобы убедиться в его работоспособности.

## Язык общения
- Отвечать на русском языке, если не указано иное.
- Документацию и комментарии писать на английском языке, если не указано иное.
- Use plain, conversational language. Avoid unexplained jargon, acronyms, and unnecessary internal details; explain any necessary technical term immediately.
- For work summaries, lead with what was done, how it was tested, what the results mean, and the recommended next step. Use short sections only when helpful, not as a mandatory template for every reply.
- Clearly distinguish completed work from plans, real checks from simulations, and observations from guarantees. Mention limitations plainly; include technical details, logs, and identifiers only when needed or requested.
