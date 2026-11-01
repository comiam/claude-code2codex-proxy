#!/usr/bin/env bash
# Claude Code statusLine for Codex quota through claude-code-proxy.
# The upstream may put the 5h or 7d window in either primary or secondary.

URL="${CODEX_USAGE_URL:-http://127.0.0.1:18765/usage}"

if ! command -v jq >/dev/null 2>&1; then
    printf 'Codex · jq not installed\n'
    exit 0
fi

# No need to read Claude Code's stdin: the session JSON is not used here.
if ! USAGE="$(curl --noproxy '*' -fsS \
    --connect-timeout 0.3 --max-time 1 "$URL" 2>/dev/null)"; then
    printf 'Codex · quota unavailable\n'
    exit 0
fi

if ! jq -e '.codex.rate_limits | type == "object"' >/dev/null 2>&1 <<<"$USAGE"; then
    printf 'Codex · quota waiting\n'
    exit 0
fi

NOW="$(date +%s)"
IFS='|' read -r LIMITED ALLOWED CAPTURED_AT <<<"$(
    jq -r '.codex as $c | $c.rate_limits as $r |
        [($r.limit_reached == true), ($r.allowed != false), ($c.captured_at_unix // "")]
        | map(tostring) | join("|")' <<<"$USAGE"
)"

# Convert a reset to an epoch. Fallback is anchored to snapshot capture time,
# rather than treating an old reset_after_seconds as a fresh countdown.
reset_epoch() {
    local at="$1" after="$2"
    if [[ "$at" =~ ^[0-9]+$ ]]; then
        printf '%s' "$at"
    elif [[ "$after" =~ ^[0-9]+$ && "$CAPTURED_AT" =~ ^[0-9]+$ ]]; then
        printf '%s' "$((CAPTURED_AT + after))"
    fi
}

remaining() {
    local seconds="$1" days hours mins
    (( seconds < 0 )) && seconds=0
    days=$((seconds / 86400))
    hours=$(((seconds % 86400) / 3600))
    mins=$(((seconds % 3600) / 60))
    if (( days > 0 )); then
        printf '%dd%dh' "$days" "$hours"
    elif (( hours > 0 )); then
        printf '%dh%02dm' "$hours" "$mins"
    else
        printf '%dm' "$mins"
    fi
}

# date uses the machine's local timezone (or its TZ environment variable).
reset_local() {
    local epoch="$1" reset_day today tomorrow
    [[ "$epoch" =~ ^[0-9]+$ ]] || { printf '?'; return; }
    today="$(date '+%Y-%m-%d')"
    tomorrow="$(date -d 'tomorrow' '+%Y-%m-%d')"
    reset_day="$(date -d "@$epoch" '+%Y-%m-%d')"
    if [[ "$reset_day" == "$today" ]]; then
        date -d "@$epoch" '+%H:%M'
    elif [[ "$reset_day" == "$tomorrow" ]]; then
        printf 'tomorrow %s' "$(date -d "@$epoch" '+%H:%M')"
    else
        date -d "@$epoch" '+%a %d %b %H:%M'
    fi
}

# Select and order quota windows by their actual duration, not by primary/secondary.
WINDOWS="$(jq -r '
    [.codex.rate_limits.primary, .codex.rate_limits.secondary]
    | map(select(type == "object" and
        (.window_minutes | type == "number") and
        (.used_percent | type == "number")))
    | sort_by([(if .window_minutes == 300 then 0
               elif .window_minutes == 10080 then 1
               else 2 end), .window_minutes])
    | .[]
    | [(.window_minutes | tostring), (.used_percent | tostring),
       (.reset_at // "" | tostring), (.reset_after_seconds // "" | tostring)]
    | join("|")
' <<<"$USAGE")"

OUT='Codex'
COUNT=0
ANY_FULL=false
NEXT_RESET=''

while IFS='|' read -r MINUTES USED RESET_AT RESET_AFTER; do
    [[ -n "$MINUTES" ]] || continue
    ((COUNT+=1))

    case "$MINUTES" in
        300)   LABEL='5h' ;;
        10080) LABEL='7d' ;;
        *)     LABEL="${MINUTES}m" ;;
    esac

    EPOCH="$(reset_epoch "$RESET_AT" "$RESET_AFTER")"
    if [[ "$EPOCH" =~ ^[0-9]+$ ]] && (( EPOCH > NOW )); then
        if [[ -z "$NEXT_RESET" ]] || (( EPOCH < NEXT_RESET )); then
            NEXT_RESET="$EPOCH"
        fi
    fi

    # An expired snapshot is not evidence of current quota exhaustion.
    if [[ "$EPOCH" =~ ^[0-9]+$ ]] && (( EPOCH <= NOW )); then
        OUT+=" · ${LABEL} awaiting refresh"
        continue
    fi

    if awk -v n="$USED" 'BEGIN { exit !(n >= 100) }'; then
        ANY_FULL=true
        OUT+=" · ${LABEL} RATE LIMITED"
        [[ -n "$EPOCH" ]] && OUT+=" · reset at $(reset_local "$EPOCH")"
    else
        LEFT="$(awk -v n="$USED" 'BEGIN {
            x = 100 - n; if (x < 0) x = 0; if (x > 100) x = 100
            printf "%.0f", x
        }')"
        OUT+=" · ${LABEL} ${LEFT}% left"
        if [[ -n "$EPOCH" ]]; then
            OUT+=" ($(remaining "$((EPOCH - NOW))"))"
        fi
    fi
done <<<"$WINDOWS"

if [[ "$LIMITED" == 'true' || "$ALLOWED" == 'false' ]] && [[ "$ANY_FULL" == 'false' ]]; then
    OUT+=' · RATE LIMITED'
    [[ -n "$NEXT_RESET" ]] && OUT+=" · next reset $(reset_local "$NEXT_RESET")"
elif (( COUNT == 0 )); then
    OUT+=' · quota waiting'
fi

printf '%s\n' "$OUT"
