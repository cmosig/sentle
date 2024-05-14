# SENTLE

Download Sentinel-2 data cubes of any scale on any machine with integrated cloud
detection, snow masking, harmonization, merging, and temporal composites. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

If you download larger areas or longer timeseries you'll need to obtain a
subscription key from [Planetary Computer](https://planetarycomputer.microsoft.com/account/request). 
Once you receive a key, which can take weeks or months set it in your shell:
`export PC_SDK_SUBSCRIPTION_KEY=xxxxyourkeyxxxx`

### Installing

```
pip install sentle
```
or 
```
git clone git@github.com:cmosig/sentle.git
cd sentle
pip install -e .
```

## Contributing

Please submit issues or pull requests if you feel like something is missing or
needs to be fixed. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thank you to [David Montero](https://github.com/davemlz) for all the
discussions and his awesome packages which inspired this.
