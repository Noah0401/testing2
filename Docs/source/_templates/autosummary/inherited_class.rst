{{ fullname.split('.')[-1].replace('.', '') }}
{{ '=' * (fullname.split('.')[-1].replace('.', '')|length) }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :special-members: __cat_dim__, __inc__
