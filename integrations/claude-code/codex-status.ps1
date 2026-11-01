# Claude Code statusLine for Codex quota through the local proxy.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding -ArgumentList $false
$separator = ' ' + [char]0x00b7 + ' '
$url = if ($env:CODEX_USAGE_URL) { $env:CODEX_USAGE_URL } else { 'http://127.0.0.1:18765/usage' }

$client = $null
$handler = $null
try {
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.UseProxy = $false
    $client = New-Object System.Net.Http.HttpClient -ArgumentList $handler
    $client.Timeout = [TimeSpan]::FromSeconds(1)
    $raw = $client.GetStringAsync($url).GetAwaiter().GetResult()
} catch {
    [Console]::WriteLine('Codex' + $separator + 'quota unavailable')
    exit 0
} finally {
    if ($client) { $client.Dispose() }
    elseif ($handler) { $handler.Dispose() }
}

try {
    $usage = ConvertFrom-Json -InputObject $raw
} catch {
    [Console]::WriteLine('Codex' + $separator + 'quota waiting')
    exit 0
}
$limits = $usage.codex.rate_limits
if ($limits -isnot [pscustomobject]) {
    [Console]::WriteLine('Codex' + $separator + 'quota waiting')
    exit 0
}

function Get-UnsignedInteger($value) {
    $number = 0L
    if ([long]::TryParse([string]$value, [ref]$number) -and $number -ge 0) {
        return $number
    }
    return $null
}

# Anchor relative resets to capture time so an old snapshot keeps its deadline.
function Get-ResetEpoch($window, $captured) {
    $at = Get-UnsignedInteger $window.reset_at
    if ($null -ne $at) { return $at }
    $after = Get-UnsignedInteger $window.reset_after_seconds
    if ($null -ne $after -and $null -ne $captured -and $after -le ([long]::MaxValue - $captured)) {
        return ($captured + $after)
    }
    return $null
}

function Format-Remaining($seconds) {
    $seconds = [Math]::Max(0, $seconds)
    $days = [Math]::Floor($seconds / 86400)
    $hours = [Math]::Floor(($seconds % 86400) / 3600)
    $mins = [Math]::Floor(($seconds % 3600) / 60)
    if ($days -gt 0) { return ('{0}d{1}h' -f $days, $hours) }
    if ($hours -gt 0) { return ('{0}h{1:00}m' -f $hours, $mins) }
    return ('{0}m' -f $mins)
}

function Format-ResetLocal($epoch) {
    try {
        $local = [DateTimeOffset]::FromUnixTimeSeconds($epoch).LocalDateTime
    } catch {
        return '?'
    }
    if ($local.Date -eq [DateTime]::Today) { return $local.ToString('HH:mm') }
    if ($local.Date -eq [DateTime]::Today.AddDays(1)) {
        return ('tomorrow ' + $local.ToString('HH:mm'))
    }
    return $local.ToString('ddd dd MMM HH:mm', [Globalization.CultureInfo]::InvariantCulture)
}

function Test-JsonNumber($value) {
    return ($value -is [int] -or $value -is [long] -or $value -is [double] -or $value -is [decimal])
}

$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$captured = Get-UnsignedInteger $usage.codex.captured_at_unix

# Window duration determines labels and ordering, not primary/secondary position.
$windows = @($limits.primary, $limits.secondary) | Where-Object {
    $_ -is [pscustomobject] -and
    (Test-JsonNumber $_.window_minutes) -and
    (Test-JsonNumber $_.used_percent)
} | Sort-Object @{
    Expression = {
        if ($_.window_minutes -eq 300) { 0 }
        elseif ($_.window_minutes -eq 10080) { 1 }
        else { 2 }
    }
}, window_minutes

$output = 'Codex'
$count = 0
$anyFull = $false
$nextReset = $null
foreach ($window in $windows) {
    $count += 1
    $label = if ($window.window_minutes -eq 300) { '5h' }
        elseif ($window.window_minutes -eq 10080) { '7d' }
        else { [string]$window.window_minutes + 'm' }
    $epoch = Get-ResetEpoch $window $captured
    if ($null -ne $epoch -and $epoch -gt $now -and ($null -eq $nextReset -or $epoch -lt $nextReset)) {
        $nextReset = $epoch
    }

    if ($null -ne $epoch -and $epoch -le $now) {
        $output += $separator + $label + ' awaiting refresh'
        continue
    }
    if ($window.used_percent -ge 100) {
        $anyFull = $true
        $output += $separator + $label + ' RATE LIMITED'
        if ($null -ne $epoch) {
            $output += $separator + 'reset at ' + (Format-ResetLocal $epoch)
        }
    } else {
        $left = [Math]::Round([Math]::Min(100, [Math]::Max(0, 100 - $window.used_percent)))
        $output += $separator + $label + ' ' + $left.ToString('0', [Globalization.CultureInfo]::InvariantCulture) + '% left'
        if ($null -ne $epoch) {
            $output += ' (' + (Format-Remaining ($epoch - $now)) + ')'
        }
    }
}

$limited = $limits.limit_reached -is [bool] -and $limits.limit_reached
$disallowed = $limits.allowed -is [bool] -and -not $limits.allowed
if (($limited -or $disallowed) -and -not $anyFull) {
    $output += $separator + 'RATE LIMITED'
    if ($null -ne $nextReset) {
        $output += $separator + 'next reset ' + (Format-ResetLocal $nextReset)
    }
} elseif ($count -eq 0) {
    $output += $separator + 'quota waiting'
}

[Console]::WriteLine($output)
