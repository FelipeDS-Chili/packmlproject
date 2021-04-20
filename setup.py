import pathlib
from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]


setup(name='packmlproject',
      version="1.1.2",
      # Below to deal with cleaner versionning
      #setup_requires=['setuptools_scm'],
      install_requires=requirements,
      #use_scm_version={'write_to': 'packmlproject/version.txt'},
      description="Package builder for Data Science projects derived from https://github.com/lologibus2/wagon_tools",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/FelipeDS-Chili/packmlproject",
      author='Felipe', author_email="falberto.inostroza@gmail.com",
      packages=find_packages(),
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/packmlproject'],
      zip_safe=False)

print( colored('############## 🔥🔥Lets make your own ML project 🔥🔥 ##############', 'green' ))
print( colored('############## Developed by Felipe Inostroza 🖥 ##############' , 'yellow' ))
print( colored('############## Base on LeWagon package ##############', 'blue' ))
