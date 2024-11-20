from setuptools import setup

setup(
    name='mmethane',
    version='0.0',
    description='Microbes and METabolites to Host Analysis Engine',
    url='http://github.com/gerberlab/mmethane',
    author='Jennifer Dawkins',
    author_email='jennifer.j.dawkins@gmail.com',
    install_requires = [
        'numpy',
    ],
    packages=['mmethane'],
    include_package_data=True,
    entry_points = {'console_scripts':
                    ['mmethane=mmethane.run']},
)