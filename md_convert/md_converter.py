print("Do not close this window until saving is complete!\n")

print("opening converter...\n")

from zmb_md_converter.gui import GUI  # noqa: E402

gui = GUI()
gui.run()
