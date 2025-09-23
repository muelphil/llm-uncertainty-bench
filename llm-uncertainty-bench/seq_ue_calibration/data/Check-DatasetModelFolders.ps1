# Root folder containing all datasets
$root = "C:\Users\PHMU3\Daten_HZDR\data_tud_capella_2025_09_22\data"   # <-- Change this to your top-level folder

# Required items (files or folder)
$requiredItems = @(
    "AnsweredCorrectly.csv",
    "AnsweredCorrectly",          # Accepts either CSV or directory
    "ClaimConditionedProbability.csv",
    "PTrueOriginal.csv",
    "Verbalized2SUEExtractor.csv"
)

# Iterate over dataset/model directories
Get-ChildItem -Path $root -Directory | ForEach-Object {
    $datasetDir = $_
    Get-ChildItem -Path $datasetDir.FullName -Directory | ForEach-Object {
        $modelDir = $_

        # Gather names of items inside the model directory
        $items = Get-ChildItem -Path $modelDir.FullName | Select-Object -ExpandProperty Name

        # Check for the special AnsweredCorrectly condition
        $hasAnsweredCorrectly = ($items -contains "AnsweredCorrectly.csv") -or ($items -contains "AnsweredCorrectly")

        # Check other required files
        $hasClaimCond   = $items -contains "ClaimConditionedProbability.csv"
        $hasPTrue       = $items -contains "PTrueOriginal.csv"
        $hasVerbalized  = $items -contains "Verbalized2SUEExtractor.csv"

        # Print if any of them are missing
        if (-not ($hasAnsweredCorrectly -and $hasClaimCond -and $hasPTrue -and $hasVerbalized)) {
            Write-Output "$($datasetDir.Name)/$($modelDir.Name)"
        }
    }
}