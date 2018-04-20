from distutils.core import setup

setup(name='SimonD3MWrapper',
    version='1.1.0',
    description='A thin client for interacting with dockerized simon primitive',
    packages=['SimonD3MWrapper'],
    install_requires=["numpy",
        "pandas",
        "requests",
        "typing",
        "Simon==1.1.0"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/simon@592cb95aadffcbaa6d87a3f14252726fee3a3ff6#egg=Simon-1.1.0"
    ],
    entry_points = {
        'd3m.primitives': [
            'distil.simon = SimonD3MWrapper:simon'
        ],
    },
)