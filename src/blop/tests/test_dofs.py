import pytest  # noqa F401

from blop.dofs import DOF, DOFList


def test_dof_types():
    dof1 = DOF(description="A continuous DOF", type="continuous", name="length", search_domain=(0, 5), units="mm")
    dof2 = DOF(
        description="A binary DOF",
        type="binary",
        name="in_or_out",
        search_domain={True, False},
    )
    dof3 = DOF(
        description="An ordinal DOF",
        type="ordinal",
        name="noise_level",
        search_domain={"low", "medium", "high"},
    )
    dof4 = DOF(
        description="A categorical DOF",
        type="categorical",
        name="fruit",
        search_domain={"mango", "banana", "papaya"},
        trust_domain={"mango", "orange", "banana", "papaya", "cantaloupe"},
    )

    dofs = DOFList([dof1, dof2, dof3, dof4])  # noqa
    dofs.__repr__()
