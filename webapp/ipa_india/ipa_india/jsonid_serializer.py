#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:03:21 2017

@author: lucadelu
"""

import django.core.serializers as serializers

JSONIDSerializer = serializers.get_serializer("json")

class Serializer(JSONIDSerializer):
    def get_dump_object(self, obj):
        data = super(Serializer, self).get_dump_object(obj)
        # Extend to your taste
        data.update(id=obj.pk)
        return data
