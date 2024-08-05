{{ fullname.split('.')[-1].replace('.', '') }}
{{ '=' * (fullname.split('.')[-1].replace('.', '')|length) }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
    .. currentmodule:: {{ module }}

    .. autoclass:: {{ objname }}
       :show-inheritance:
       :members:
       :inherited-members:
       :special-members: __cat_dim__, __inc__
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. currentmodule:: {{ module }}

    .. autoclass:: {{ objname }}
       :show-inheritance:
       :members:
       :inherited-members:
       :special-members: __cat_dim__, __inc__
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
    .. currentmodule:: {{ module }}

    .. autoclass:: {{ objname }}
       :show-inheritance:
       :members:
       :inherited-members:
       :special-members: __cat_dim__, __inc__
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
