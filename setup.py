import setuptools

setuptools.setup(
    entry_points={
        'aiida.workflows': [
            'ispg.conformer_opt = aiidalab_ispg.workflows:ConformerOptimizationWorkChain',
            'ispg.atmospec = aiidalab_ispg.workflows:AtmospecWorkChain',
            'ispg.wigner = aiidalab_ispg.workflows:generate_wigner_structures'
        ]
    }
)

