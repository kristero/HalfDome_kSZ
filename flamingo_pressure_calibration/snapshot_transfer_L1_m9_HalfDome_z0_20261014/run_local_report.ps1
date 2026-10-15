# Guard only this study's local process; cluster calculations are separate.
param([ValidateSet('report','preview','qa')][string]$Task = 'report')
$ErrorActionPreference = 'Stop'
$study = '/home/cbllover/HalfDome/flamingo_pressure_calibration/snapshot_transfer_L1_m9_HalfDome_z0_20261014'
$windowsStudy = '\\wsl.localhost\myrootfs' + $study.Replace('/', '\')

function Get-FreeMemoryGiB {
    $os = Get-CimInstance Win32_OperatingSystem
    $windowsGiB = $os.FreePhysicalMemory / 1MB
    $line = wsl.exe -d myrootfs --exec awk '/^MemAvailable:/ {print $2}' /proc/meminfo
    $linuxGiB = [double]$line / 1MB
    return [math]::Min($windowsGiB, $linuxGiB)
}

$script = switch ($Task) {
    'report' {'make_reports.py'}
    'preview' {'plot_measured_preview.py'}
    'qa' {'verify_final_reports.py'}
}
if ((Get-FreeMemoryGiB) -lt 8) {
    throw 'Less than 8 GiB is available. Keep 5 GiB free plus working headroom; the local process was not started.'
}
$worker = Start-Process -FilePath 'wsl.exe' -WindowStyle Hidden -PassThru `
    -ArgumentList @('-d','myrootfs','--cd',$study,'--exec',
                    '/home/cbllover/miniconda3/bin/python',"$study/$script") `
    -RedirectStandardOutput "$windowsStudy/local_${Task}_build.log" `
    -RedirectStandardError "$windowsStudy/local_${Task}_build_errors.log"
$processHandle = $worker.Handle

while (-not $worker.HasExited) {
    Start-Sleep -Seconds 3
    $freeGiB = Get-FreeMemoryGiB
    if ($freeGiB -lt 5) {
        Write-Warning "Available memory is $freeGiB GiB. Stopping this study's PDF process."
        $pidFile = "$windowsStudy/$Task.pid"
        if (Test-Path -LiteralPath $pidFile) {
            $reportPid = [int](Get-Content -LiteralPath $pidFile)
            wsl.exe -d myrootfs --exec /bin/kill -TERM $reportPid
        }
        if (-not $worker.HasExited) { Stop-Process -Id $worker.Id -Force }
        if ($freeGiB -lt 2) {
            Write-Warning 'Critical memory threshold: below 2 GiB.'
            # Only this study's regenerable PDF intermediates may be removed.
            $temporary = "$windowsStudy/tmp/pdfs"
            if (Test-Path -LiteralPath $temporary) {
                $resolvedStudy = (Resolve-Path -LiteralPath $windowsStudy).ProviderPath
                $resolvedTemp = (Resolve-Path -LiteralPath $temporary).ProviderPath
                if (-not $resolvedTemp.StartsWith($resolvedStudy + '\')) {
                    throw 'Temporary path is outside this study.'
                }
                Get-ChildItem -LiteralPath $resolvedTemp -File | Remove-Item -Force
            }
        }
        throw 'PDF generation stopped to preserve at least 5 GiB of available memory.'
    }
    $worker.Refresh()
}
$worker.WaitForExit()
if ($worker.ExitCode -ne 0) { throw "Local $Task generation failed: exit $($worker.ExitCode)." }
Get-Content -LiteralPath "$windowsStudy/local_${Task}_build.log"
