from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.1.1',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy",
        "pandas",
        "requests",
        "typing",
        "Simon==1.2.0"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/simon@77d1bfe45d52a41918dad551efa7eab619a9ca9a#egg=Simon-1.2.0"
    ],
    entry_points = {
        'd3m.primitives': [
            'distil.simon = SimonD3MWrapper:simon'
        ],
    },
)