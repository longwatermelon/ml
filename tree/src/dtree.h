#pragma once
#include <memory>
#include <vector>

namespace dtree
{
    class DTree
    {
    public:
        DTree();
        ~DTree();

    private:
        std::unique_ptr<DTree> m_no, m_yes;
        int feature{ 0 };
    };
}
