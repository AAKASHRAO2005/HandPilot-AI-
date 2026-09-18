' HandPilot AI — Silent Background Launcher (No cmd console)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Check if venv pythonw exists
Set fso = CreateObject("Scripting.FileSystemObject")
If fso.FileExists(".venv\Scripts\pythonw.exe") Then
    WshShell.Run """.venv\Scripts\pythonw.exe"" app.py", 0, False
ElseIf fso.FileExists("venv\Scripts\pythonw.exe") Then
    WshShell.Run """venv\Scripts\pythonw.exe"" app.py", 0, False
Else
    WshShell.Run "pythonw.exe app.py", 0, False
End If
