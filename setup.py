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
        "git+https://github.com/NewKnowledge/simon@c9f4ebad916028629096400de850410ad17aaaa1#egg=Simon-1.2.0"
    ],
    entry_points = {
        'd3m.primitives': [
            'distil.simon = SimonD3MWrapper:simon'
        ],
    },
)