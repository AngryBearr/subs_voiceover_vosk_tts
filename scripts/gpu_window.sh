#!/usr/bin/env bash
# gpu_window.sh <log> <command...>: run a GPU experiment while the season run is not transcribing
# (ASR / diarization / OCR take most of the card); stop it when the next episode's transcription starts.
LOG=$1; shift
busy() { grep -E "stt.run: \[[^]]*\] [a-z]+ \.\.\." "$LOG" | tail -1 | grep -q "\] transcribe \.\.\."; }
while busy; do sleep 20; done
start=$(grep -cE "\] transcribe \.\.\." "$LOG")
"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  if [ "$(grep -cE "\] transcribe \.\.\." "$LOG")" -gt "$start" ]; then
    echo "=== $(date '+%T') transcription started: stopping $pid"; kill $pid; wait $pid; exit 75
  fi
  sleep 5
done
wait $pid
