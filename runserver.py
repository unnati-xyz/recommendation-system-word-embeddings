from sherlock.api import app


host="localhost"
port=5000
debug=True

print("starting server on host %s port %s"%(host, port))
app.run(host=host, port=port, debug=debug)
