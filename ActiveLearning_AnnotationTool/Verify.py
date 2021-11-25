t = [
["ActiveLearning.py", "ef5415977e9633470b331af5c5b65016"],
["ActiveLearningLayout.py", "49e5ce1c65601cd5317aabc4b6ad2c11"],
["ActiveLearningOG.py", "1d7d1f2236bfa3e7019438dcdd687f30"],
["ActivelearningOutputManager.py", "00b7f9e580a5711b7a6572080ad4d38b"],
["ALData/find.py", "78bb1394c9cb91f268b94bd019b4e830"],
["ALData/__MACOSX/Backend/._AL-backend.py", "0cc7aaa1b1cef2d7721d3f3a1440c1e0"],
["Bridge.py", "2e66524f27407ab48fc941d9a1496b68"],
["Config.py", "222022613a6a4063bc6fbef4e9b232fa"],
["DbgServer.py", "5d19a6177db8ff52253b8e9c38e33165"],
["DebugSenseViewBreak.py", "a873863020958448c79a70dcf8f406fa"],
["findPath.py", "cf69dd2b50597034807a9f321f70b43b"],
["Layouts/__init__.py", "94212f06052bae6312175b6a623cc918"],
["ND/FrameBar.py", "5ab0c51c94607e799a69dbb1145ff271"],
["ND/FrameBarTest.py", "d518bbc2d4e5a0baf22bf8a988f1192a"],
["ND/Ui_FrameBarTestDialog.py", "5aabf968e7e1c567b50ed5610418f881"],
["OutputHook.py", "3ecc61e2b105379fc3f5b9e8b48fefb6"],
["ProcessingWindow.py", "6bfb4be56721a9fd4c9ef890f32ca4fc"],
["SenseViewBreak.py", "70da8e30d73c55973a7d649661abaa4a"],
["Sensors/VideoSensor.py", "026d014d2596ec69b0d16ecf89085d2c"],
["TimeAxis.py", "a31907a09d1504253786eeb11aab52e6"],
["Timeline.py", "978d9f3d868ec33aa06a7058ff5cb586"],
["Tracer.py", "5aaaaf6f3bad057fab302377f6d07599"],
["ui_ActiveLearning.py", "3ef0e4c903b54f9abc9b35740c32897b"],
["ui_Timeline.py", "e357d16b913131a8f15ab8089fffdc6c"],
["Util.py", "25676ed01879e1fba72c229cf0aacacc"],
["Widgets/FrameBarWidget.py", "7215ee9c7d9dc229d2921a40e899ec5f"],
["Widgets/SakuraPlotWidget.py", "a46625f5645cc0c042ef79372d9c5349"],
["Widgets/VideoWidget.py", "a76c623d7d6fe017fb28dd8d2c2be8d7"],
["Widgets/__init__.py", "d41d8cd98f00b204e9800998ecf8427e"],
]

import hashlib
def get_md5(file_path):
    with open(file_path, 'rb') as file_handle:
        md5 = hashlib.md5()
        md5.update(file_handle.read())
        return md5.hexdigest()



for tt in t:
    file_path = tt[0]
    file_md5 = tt[1]

    result_str = ("OK" if get_md5(file_path) == file_md5 else "ERROR")

    print(f'Verify file: {file_path} ... [{get_md5(file_path)}] ... '.ljust(90, " ") + result_str)


