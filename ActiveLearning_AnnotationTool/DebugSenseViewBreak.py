

from Tracer import *

from SenseViewBreak import *



if __name__ == '__main__':
    app = QApplication([])

    args = parser.parse_args()

    active_learning = ActiveLearning()
    
    timeline = Timeline()
    window = SenseView(args, timeline)
    window.show()
    window.timeline.attach_to(window)
    active_learning.attach_to(window)
    window.setup_al_window(active_learning)
    active_learning.show()
    window.activateWindow()
    app.exit(app.exec_())
