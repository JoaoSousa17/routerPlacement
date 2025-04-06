
Make sure you have `pipenv` installed: 

```bash
$ pip install --user pipenv
```

Then make sure your `cwd` is the root of the project and run

```bash
$ pipenv install
```

After that you can use the repo. For it to work properly, you have to invoke python
through pipenv. There are two ways:

```bash
$ pipenv run python ...
```

or you can do 

```bash
$ pipenv shell
```

and from now on running python will work (make sure you don't close the shell, otherwise
you have to run the command again). If you use VSCode, it has support for pipenv, but
you will have to figure that one on your own.

**To run the project code** do (maybe you will have to prefix `pipenv run`)

```bash
$ python -m routers <path-to-input> [...arguments]
```
    
