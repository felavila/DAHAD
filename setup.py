from setuptools import setup

setup(
    name='DAHAD',
    version='0.0.1',
    description="Download and Handling of Astronomical Data (DAHAD)",
    url='?',
    author='?',
    author_email='?',  # Optional: add if you want to display a contact
    license='CC0 1.0 Universal (Public Domain Dedication)',
    packages=['DAHAD'],
    install_requires=["pandas","astroquery","sparclclient","astro-datalab"],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)

