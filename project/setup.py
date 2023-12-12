from setuptools import find_packages, setup
from glob import glob

package_name = 'project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/crosswalk*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tufourn',
    maintainer_email='tufourn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect=project.detect:main',
            'stopsignal = project.stopsignal:main'
        ],
    },
)
