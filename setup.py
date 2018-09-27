from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.1.2',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy",
        "pandas",
        "requests",
        "typing",
        "Simon==1.2.1"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/simon@674cac9a96c93b20be0eddfc517ada56472c9a22#egg=Simon-1.2.1"
    ],
    entry_points = {
        'd3m.primitives': [
            'distil.simon = SimonD3MWrapper:simon'
        ],
    },
)