from setuptools import setup, find_packages

setup(
    name='mmethane',
    version='0.2',
    description='Microbes and METabolites to Host Analysis Engine',
    url='http://github.com/gerberlab/mmethane',
    author='Jennifer Dawkins',
    author_email='jennifer.j.dawkins@gmail.com',
    install_requires = [
    ],
    packages=["mmethane", "mmethane.utilities"],
    include_package_data=True,
    entry_points = {'console_scripts':
                    ['mmethane=mmethane.run_package:run']},
    python_requires=">=3.6"
)
