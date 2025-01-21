import pytest  # noqa F401

from blop.dofs import DOF, DOFList


def test_dof_types():
    dof1 = DOF(description="A continuous DOF", type="continuous", name="x1", search_domain=[0, 5], units="mm")
    dof2 = DOF(
        description="A binary DOF",
        type="binary",
        name="x2",
        search_domain={"in", "out"},
        units="is it in or out?",
    )
    dof3 = DOF(
        description="An ordinal DOF",
        type="ordinal",
        name="x3",
        search_domain={"low", "medium", "high"},
        units="noise level",
    )
    dof4 = DOF(
        description="A categorical DOF",
        type="categorical",
        name="x4",
        search_domain={"mango", "orange", "banana", "papaya"},
        units="fruit",
    )

    dofs = DOFList([dof1, dof2, dof3, dof4])  # noqa
