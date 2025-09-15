from webapp import create_app

app = create_app()

@app.route("/_ls_static")
def _ls_static():
    import os
    root = app.static_folder
    try:
        items = os.listdir(root)
    except Exception as e:
        return f"static_folder: {root} | ERROR: {e}"
    return "static_folder: " + root + "<br>" + "<br>".join(items)


if __name__ == "__main__":
    app.run(debug=True)
