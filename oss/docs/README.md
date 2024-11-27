# jaxloop ReadTheDocs Site

To build:

```
# Install Sphinx deps.
pip install -r third_party/py/jaxloop/oss/docs/requirements.txt

# Temporary step necessary for Python project install
cp third_party/py/jaxloop/*.py third_party/py/jaxloop/oss/jaxloop
mkdir third_party/py/jaxloop/oss/jaxloop/step_number_writer
cp third_party/py/jaxloop/step_number_writer/*.py third_party/py/jaxloop/oss/jaxloop/step_number_writer

# Install as external package.
pip install -e third_party/py/jaxloop/oss/

make -C third_party/py/jaxloop/oss/docs html

# Preview locally
python -m http.server -d /tmp/jaxloop_docs/html

# Preview in google3
cp -R /tmp/jaxloop_docs/html/* /google/data/rw/users/je/jeffcarp/www/jaxloop_docs
```
