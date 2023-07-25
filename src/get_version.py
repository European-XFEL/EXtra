from pathlib import Path

from setuptools_scm import Configuration, ScmVersion, _do_parse, dump_version


def _custom_version(v: ScmVersion):
    version = str(v.tag)

    if v.exact:
        return version

    if v.distance:
        version += f".{v.distance}"

    if v.node:
        # Drop off the "g" prefix
        short_hash = v.node[1:]
        version += f"+{short_hash}"

    return version


config = Configuration(root=".")

v = _do_parse(config)

if v is None:
    raise RuntimeError("No version was found")


version = _custom_version(v)


dump_version(
    root=config.root,
    version=version,
    write_to=Path(__file__).parent / "extra" / "__version__.py",
    template=config.write_to_template,
)
